import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import { Business } from "@/types/BusinessTypes";
import { Metadata } from "next";
export const metadata: Metadata = {
    title: 'For You',
    description: '...',
}

export default function forYouPage() {
    //const MOCK_DATA: Business[] = ;


    return (
        <div className="bg-[#f4f4f4]">
            <div className="flex justify-center items-center min-h-screen">
                {/**<BusinessExplore businesses={MOCK_DATA} />**/}
            </div>
        </div>
    );
}