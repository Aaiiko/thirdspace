
import { Business } from "@/types/BusinessTypes";

export const BusinessCard = (business: Business) => {

    /** For rendering our stars grin */
    const renderStars = (rating: number) => {
        const stars = [];
        for (let i = 1; i <= 5; i++) {
            stars.push(
                <span key={i} className={i <= rating ? "text-yellow-500" : "text-gray-300"}>
                    ★
                </span>
            );
        }
        return stars;
    };

        /** For rendering our grin */
        const renderDollar = (rating: number) => {
            const stars = [];
            for (let i = 1; i <= rating; i++) {
                stars.push(
                    <span key={i} className="font-bold">
                        $
                    </span>
                );
            }
            return stars;
        };

    return (
        <div className="bg-[#f4f4f4] rounded-lg w-full" key={business.id} style={{ width: "100%" }} >
            <img
                src={business.image ? business.image : '/Phantom.svg'}
                alt={business.name}
                className="w-full object-cover"
                style={{ height: "600px" }}
                onError={(e) => { e.currentTarget.src = '/Phantom.svg' }}
            />
            <div className="px-1 py-2">
                <div>
                    <h2 className="text-3xl font-bold text-[#365b6d]">{business.name}</h2>
                    <p className="font-bold text-md text-[#8ca9ad]">{business.location}</p>
                </div>
                <p className="font-work-sans-regular text-[#8ca9ad]">Category {business.category}</p>
                <div className="flex items-center mb-4">
                    {renderStars(business.review)}
                    <span className="ml-2 text-gray-600 text-[#8ca9ad]">{business.review}</span>
                </div>
                <div className="flex items-center mb-4">
                    {renderDollar(business.review)}
                    <span className="ml-2 text-gray-600 text-[#8ca9ad]">{business.review}</span>
                </div>
                <p className="text-xl text-[#365b6d] font-bold">About</p>
                <p className="text-[#8ca9ad] mb-6">{business.description}</p>
                {/* {business.tags.map(tag => (
                    <p className="text-[#8ca9ad] mb-6">{tag}</p>
                ))} */}
                
            </div>

        </div>
    );
};